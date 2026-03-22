export interface BasicBeneficiaryInfo {

    account: number;
    beneficiaryName: string;
    beneficiaryEmail: string;

}

export class BasicBeneficiaryInfoModel implements BasicBeneficiaryInfo {

    constructor(
        public account: number,
        public beneficiaryName: string,
        public beneficiaryEmail: string
    ) { }
}